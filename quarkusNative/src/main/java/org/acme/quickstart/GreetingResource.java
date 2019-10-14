package org.acme.quickstart;

import java.io.InputStream;
import java.util.List;

import javax.ws.rs.Consumes;
import javax.ws.rs.HeaderParam;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.PathParam;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MultivaluedMap;

import org.jboss.resteasy.plugins.providers.multipart.InputPart;
import org.jboss.resteasy.plugins.providers.multipart.MultipartFormDataInput;

@Path("/quarkusai")
public class GreetingResource {

    @POST
    @Path("/labelImageNative/{results}")
    @Consumes("multipart/form-data")
    @Produces("application/json")
    public ImageProcessingResult loadImage(@HeaderParam("Content-Length") String contentLength, @PathParam("results") int results, MultipartFormDataInput input) throws Exception {
        long before = System.currentTimeMillis();
        InputPart inputPart = input.getFormDataMap().get("file").iterator().next();
        String fileName = parseFileName(inputPart.getHeaders());
        List<Probability> probs = LabelImage.labelImage(fileName, inputPart.getBody(InputStream.class, null)).subList(0, results);
        return new ImageProcessingResult((System.currentTimeMillis() - before), probs);
    }

    // Parse Content-Disposition header to get the original file name
    private static String parseFileName(MultivaluedMap<String, String> headers) {
        String[] contentDispositionHeader = headers.getFirst("Content-Disposition").split(";");
        for (String name : contentDispositionHeader) {
            if ((name.trim().startsWith("filename"))) {
                String[] tmp = name.split("=");
                String fileName = tmp[1].trim().replaceAll("\"", "");
                return fileName;
            }
        }
        return "randomName";
    }
}
